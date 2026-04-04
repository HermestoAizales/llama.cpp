<script lang="ts">
	import { ChevronDown, ChevronUp } from '@lucide/svelte';

	export interface TokenLogprobData {
		token: string;
		logprob: number;
		top_logprobs: Array<{ token: string; logprob: number }>;
	}

	interface Props {
		tokens: TokenLogprobData[];
	}

	let { tokens }: Props = $props();
	let expanded = $state(false);

	function computeEntropy(data: TokenLogprobData): number {
		let h = 0;
		const items =
			data.top_logprobs.length > 0
				? data.top_logprobs
				: [{ token: data.token, logprob: data.logprob }];
		for (const t of items) {
			const p = Math.exp(t.logprob);
			if (p > 0 && p <= 1) {
				h -= p * Math.log2(p);
			}
		}
		return h;
	}

	function formatToken(t: string): string {
		return t.replace(/\n/g, '\\n').replace(/\t/g, '\\t');
	}

	let displayTokens = $derived(tokens.slice(0, 100));
	let hasMore = $derived(tokens.length > 100);
</script>

<div class="mt-2 rounded-md border border-border bg-muted/30">
	<button
		onclick={() => (expanded = !expanded)}
		class="flex w-full items-center justify-between rounded-md px-3 py-2 text-sm hover:bg-muted/50"
	>
		<div class="flex items-center gap-2">
			{#if expanded}
				<ChevronDown class="h-4 w-4" />
			{:else}
				<ChevronUp class="h-4 w-4" />
			{/if}
			<span class="font-medium">Token Inspection</span>
			<span class="text-xs text-muted-foreground">
				{tokens.length} tokens
			</span>
		</div>
	</button>

	{#if expanded}
		<div class="max-h-96 space-y-1 overflow-y-auto px-3 pb-3">
			{#each displayTokens as t, idx (t.token + idx)}
				{@const items =
					t.top_logprobs.length > 0 ? t.top_logprobs : [{ token: t.token, logprob: t.logprob }]}
				{@const mainProb = Math.exp(t.logprob)}
				{@const entropy = computeEntropy(t)}
				{@const probClass =
					mainProb >= 0.7
						? 'bg-green-500 text-white'
						: mainProb >= 0.3
							? 'bg-yellow-500 text-white'
							: 'bg-red-500 text-white'}
				<div
					class="cursor-default rounded border border-border/50 bg-background/50 px-2 py-1.5 text-xs transition-colors hover:bg-muted/30"
				>
					<div class="mb-1 flex items-center gap-2">
						<span class="w-7 text-right font-mono font-bold text-muted-foreground select-none">
							#{idx + 1}
						</span>
						<span class="max-w-[120px] truncate font-mono font-semibold" title={t.token}>
							{formatToken(t.token)}
						</span>
						<span
							class="inline-flex items-center rounded px-1.5 py-0.5 text-xs font-medium tabular-nums {probClass}"
						>
							{(mainProb * 100).toFixed(1)}%
						</span>
						<span class="ml-auto text-muted-foreground tabular-nums">
							logprob: {t.logprob.toFixed(3)}
						</span>
						<span
							class="font-medium tabular-nums"
							class:text-green-500={entropy < 1}
							class:text-yellow-500={entropy >= 1 && entropy < 3}
							class:text-red-500={entropy >= 3}
						>
							H={entropy.toFixed(2)}
						</span>
					</div>

					{#if items.length > 1}
						<div class="mt-1 space-y-0.5 border-t border-border/30 pt-1">
							{#each items.slice(0, 5) as alt, ai (alt.token + ai)}
								{@const altProb = Math.exp(alt.logprob)}
								<div class="flex items-center gap-1">
									<span class="w-3 text-right text-[10px] text-muted-foreground">{ai + 1}.</span>
									<span class="w-16 truncate font-mono text-[11px]" title={alt.token}>
										{formatToken(alt.token)}
									</span>
									<div class="h-3 flex-1 overflow-hidden rounded bg-muted/40">
										<div
											class="h-full rounded-sm"
											class:bg-green-500={altProb >= 0.7}
											class:bg-yellow-500={altProb >= 0.3 && altProb < 0.7}
											class:bg-red-500={altProb < 0.3}
											style={'width: ' + Math.min(altProb * 100, 100) + '%;'}
										></div>
									</div>
									<span class="w-10 text-right text-[10px] text-muted-foreground tabular-nums">
										{(altProb * 100).toFixed(1)}%
									</span>
								</div>
							{/each}
							{#if items.length > 5}
								<span class="block text-right text-[10px] text-muted-foreground">
									+{items.length - 5} more
								</span>
							{/if}
						</div>
					{/if}
				</div>
			{/each}
			{#if hasMore}
				<div class="py-1 text-center text-xs text-muted-foreground">
					Showing first 100 of {tokens.length} tokens
				</div>
			{/if}
		</div>
	{/if}
</div>
